import {request} from './client';import {Run} from '../types/domain';
export type RunMode='single'|'multi';
export const getRuns=()=>request<Run[]>('/runs');
export const createRun=(data:{name:string;study_id:number;alpha:number;seed:number;mode:RunMode})=>request<Run>('/runs',{method:'POST',body:JSON.stringify(data)});

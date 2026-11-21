module class6_tree23(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if((i[193] & (~i[110])) | ((~i[193]) & (~i[110])) | ((~i[353]) & i[149] & i[110]) | (i[173] & (~i[149]) & i[110]) | ((~i[173]) & (~i[149]) & i[110]) | (i[338] & i[169] & i[353] & i[149] & i[110]) | (i[48] & (~i[169]) & i[353] & i[149] & i[110]) | ((~i[48]) & (~i[169]) & i[353] & i[149] & i[110]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
